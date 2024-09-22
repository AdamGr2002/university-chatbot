import React from 'react';
import { useClerk } from '@clerk/clerk-react';

function SignOutButton() {
  const { signOut } = useClerk();

  return (
    <button
      onClick={() => signOut()}
      className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 transition duration-150 ease-in-out"
    >
      Sign Out
    </button>
  );
}

export default SignOutButton;
